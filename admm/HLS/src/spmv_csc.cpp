#include "spmv_csc.h"

#include "ap_int.h"

static constexpr int NNZ_LANES = (NUM_PES < PACK_SIZE) ? NUM_PES : PACK_SIZE;

struct NnzPack
{
    int row[NNZ_LANES];
    float val[NNZ_LANES];
    ap_uint<8> n;
};

struct NnzPkt
{
    int row;
    float val;
    float x;
    bool last;
};

struct ColInfo
{
    int nnz;
    float x;
};

/*
 * Reads column pointer information and corresponding x values, and sends them as a stream of ColInfo structs.
 * Each ColInfo contains the number of non-zeros in the column and the corresponding x value
 */
static void read_col_info(int num_cols,
                          const int *A_col_ptr,
                          const float *x_in,
                          hls::stream<ColInfo> &col_info_stream)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable = x_in type = cyclic factor = RESHAPE_FACTOR dim = 1
    int prev = A_col_ptr[0];

    float x_element = x_in[0];

    for (int col = 0; col < num_cols - 1; ++col)
    {
#pragma HLS PIPELINE II = 1
        int next = A_col_ptr[col + 1];

        int col_nnz = next - prev;

        if (col_nnz > 0)
        {
            ColInfo col_info;
            col_info.nnz = col_nnz;
            col_info.x = x_element;
            col_info_stream.write(col_info);
        }
        prev = next;
        x_element = x_in[col + 1];
    }

    // Handle last column separately to avoid excessive logic in the loop
    int next = A_col_ptr[num_cols];
    int col_nnz = next - prev;
    if (col_nnz > 0)
    {
        ColInfo col_info;
        col_info.nnz = col_nnz;
        col_info.x = x_element;
        col_info_stream.write(col_info);
    }
}

/*
 * Read A_row_idx and A_values in groups of 16 (PACK_SIZE) in bursts.
 * Each burst is packed into NnzPack structs containing up to NNZ_LANES valid entries.
 */
static void read_nnz_packed(int nnz,
                            const int16 *row_in,
                            const float16 *val_in,
                            hls::stream<NnzPack> &nnz_stream)
{
#pragma HLS INLINE off
    const int words = CEIL_DIV(nnz, PACK_SIZE);
    int idx = 0;

    for (int w = 0; w < words; ++w)
    {
        int16 rows = row_in[w];
        float16 vals = val_in[w];

        for (int base = 0; base < PACK_SIZE; base += NNZ_LANES)
        {
#pragma HLS PIPELINE II = 1
            if (idx >= nnz)
                continue;

            // How many nnz remain
            int count = nnz - idx;
            // How many elements are left in the current word starting from base
            const int max_in_word = PACK_SIZE - base;
            // We can only pack up to NNZ_LANES at a time
            const int max_this_pack = (max_in_word < NNZ_LANES) ? max_in_word : NNZ_LANES;
            if (count > max_this_pack)
            {
                // How many elements are being packed in this iteration
                count = max_this_pack;
            }
            NnzPack pack;
            pack.n = (ap_uint<8>)count;

            for (int i = 0; i < NNZ_LANES; ++i)
            {
#pragma HLS UNROLL
                const bool valid_lane = (i < count);
                if (valid_lane)
                {
                    pack.row[i] = rows[base + i];
                    pack.val[i] = vals[base + i];
                }
                else
                {
                    pack.row[i] = 0;
                    pack.val[i] = 0.0f;
                }
            }

            nnz_stream.write(pack);
            idx += count;
        }
    }
}

/*
 * Distributes nnz elements to PEs in a continuous round-robin way.
 * Each cycle, up to NNZ_LANES elements are dispatched in parallel to the PE streams,
 * reading column info and nnz packs.
 *
 * Nnz elements are sent in NNZ_LANES groups. If the current column has fewer than NNZ_LANES
 * remaining, only that many are sent and the next column's info is read in the same cycle. The next
 * elements will be sent to the next PEs in the round-robin order. This ensures that all PEs receive
 * work as soon as possible without waiting for an entire column to be dispatched.
 *
 * When all nnz have been dispatched, termination packets with last=true are sent to each PE.
 */
static void distribute_to_pe(int nnz,
                             hls::stream<ColInfo> &col_info_stream,
                             hls::stream<NnzPack> &nnz_stream,
                             hls::stream<NnzPkt> pe_streams[NUM_PES])
{
#pragma HLS INLINE off
    // Which PE is next in line to receive work
    int pe = 0;

    NnzPack pack;
#pragma HLS ARRAY_PARTITION variable = pack.row complete
#pragma HLS ARRAY_PARTITION variable = pack.val complete
    // Read position in the current pack
    int pack_pos = 0;
    // Total nnz elements in the current pack
    int pack_n = 0;

    // Remaining nnz in the current column
    int col_nnz = 0;
    // Value of x for the current column
    float x_cur = 0.0f;
    // Total nnz distributed so far (for termination)
    int distributed = 0;

    while (distributed < nnz)
    {
#pragma HLS PIPELINE II = 1

        // Start working on next column
        if (col_nnz == 0)
        {
            ColInfo info = col_info_stream.read();
            col_nnz = info.nnz;
            x_cur = info.x;
        }
        // If current pack is finished, read the next one
        if (pack_pos >= pack_n)
        {
            pack = nnz_stream.read();
            pack_n = (int)pack.n;
            pack_pos = 0;
        }

        int available = pack_n - pack_pos;
        // Only send number of nnz available in the current pack that belong to the current column and fit in the PE streams
        int send = (available < col_nnz) ? available : col_nnz;
        if (send > NNZ_LANES)
            send = NNZ_LANES;

        // Unrolled for each PE
        // Write one nnz element to each PE stream in a round-robin way until we've sent 'send' elements
        for (int p = 0; p < NUM_PES; ++p)
        {
#pragma HLS UNROLL
            int rel = p - pe;
            if (rel < 0)
                rel += NUM_PES;

            if (rel < send)
            {
                NnzPkt pkt;
                pkt.row = pack.row[pack_pos + rel];
                pkt.val = pack.val[pack_pos + rel];
                pkt.x = x_cur;
                pkt.last = false;
                pe_streams[p].write(pkt);
            }
        }
        // Move the current PE pointer forward by 'send' positions in a round-robin way
        pe += send;
        // Wrap around if we exceed the number of PEs
        if (pe >= NUM_PES)
        {
            pe -= NUM_PES;
        }

        // Update counters
        pack_pos += send;
        col_nnz -= send;
        distributed += send;
    }

    // After all nnz have been distributed, send termination packets to each PE
    for (int i = 0; i < NUM_PES; ++i)
    {
#pragma HLS PIPELINE II = 1
        NnzPkt end;
        end.row = 0;
        end.val = 0.0f;
        end.x = 0.0f;
        end.last = true;
        pe_streams[i].write(end);
    }
}

static void compute_pe(int num_rows, hls::stream<NnzPkt> &in, float y_partial[MAX_ROWS], bool clear_y)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = y_partial type = cyclic factor = RESHAPE_FACTOR dim = 1

    if (clear_y)
    {
        for (int i = 0; i < num_rows; i += RESHAPE_FACTOR)
        {
#pragma HLS PIPELINE II = 1
            for (int j = 0; j < RESHAPE_FACTOR; ++j)
            {
#pragma HLS UNROLL
                const int idx = i + j;
                if (idx < num_rows)
                {
                    y_partial[idx] = 0.0f;
                }
            }
        }
    }

    while (true)
    {
#pragma HLS PIPELINE II = 1
        NnzPkt pkt = in.read();
        if (pkt.last)
            break;

        int row = pkt.row;
        if ((unsigned)row < (unsigned)num_rows)
        {
            y_partial[row] += pkt.val * pkt.x;
        }
    }
}

static void reduce_and_accumulate(int num_rows,
                                  float y_partial[NUM_PES][MAX_ROWS],
                                  float *y_out)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable = y_out type = cyclic factor = RESHAPE_FACTOR dim = 1
    for (int i = 0; i < num_rows; i += RESHAPE_FACTOR)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < RESHAPE_FACTOR; ++j)
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < num_rows)
            {
                float sum = 0.0f;
                for (int pe = 0; pe < NUM_PES; ++pe)
                {
#pragma HLS UNROLL
                    sum += y_partial[pe][idx];
                }
                y_out[idx] = sum;
            }
        }
    }
}

static void spmv_csc_dataflow(int num_rows,
                              int num_cols,
                              int nnz,
                              const int16 *A_row_idx,
                              const int *A_col_ptr,
                              const float16 *A_values,
                              const float *x,
                              float y_partial[NUM_PES][MAX_ROWS],
                              bool clear_y)
{
#pragma HLS INLINE off

    hls::stream<ColInfo> col_info_stream("col_info_stream");
    hls::stream<NnzPack> nnz_stream("nnz_stream");
    hls::stream<NnzPkt> pe_streams[NUM_PES];

#pragma HLS STREAM variable = col_info_stream depth = 64
#pragma HLS STREAM variable = nnz_stream depth = 256
#pragma HLS STREAM variable = pe_streams depth = 256

#pragma HLS DATAFLOW
    read_col_info(num_cols, A_col_ptr, x, col_info_stream);
    read_nnz_packed(nnz, A_row_idx, A_values, nnz_stream);
    distribute_to_pe(nnz, col_info_stream, nnz_stream, pe_streams);

    for (int pe = 0; pe < NUM_PES; ++pe)
    {
#pragma HLS UNROLL
        compute_pe(num_rows, pe_streams[pe], y_partial[pe], clear_y);
    }
}

void spmv_csc(int num_rows,
              int num_cols,
              int nnz,
              const int16 *A_row_idx,
              const int *A_col_ptr,
              const float16 *A_values,
              const float *x,
              float *y,
              bool clear_y,
              bool write_y)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable = x type = cyclic factor = RESHAPE_FACTOR dim = 1
#pragma HLS ARRAY_RESHAPE variable = y type = cyclic factor = RESHAPE_FACTOR dim = 1

    if (num_rows > MAX_ROWS || num_cols > MAX_COLS)
        return;

    static float y_partial[NUM_PES][MAX_ROWS];

#pragma HLS ARRAY_PARTITION variable = y_partial complete dim = 1
#pragma HLS ARRAY_PARTITION variable = y_partial type = cyclic factor = RESHAPE_FACTOR dim = 2
#pragma HLS BIND_STORAGE variable = y_partial type = ram_t2p impl = bram

    spmv_csc_dataflow(num_rows, num_cols, nnz, A_row_idx, A_col_ptr, A_values, x, y_partial, clear_y);

    if (write_y)
    {
        reduce_and_accumulate(num_rows, y_partial, y);
    }
}

void spmv_csc_tiled(int global_num_rows,
                    int global_num_cols,
                    const TiledMatrix &A,
                    float *x,
                    float *y)
{
    // Process one row-tile at a time so `spmv_csc`'s internal accumulator
    // can accumulate across all column-tiles for that tile row
    for (int tile_row = 0; tile_row < A.num_row_tiles; ++tile_row)
    {
        int row_base = tile_row * MAX_ROWS;
        int rows_this_tile = (row_base + MAX_ROWS <= global_num_rows) ? MAX_ROWS : (global_num_rows - row_base);

        for (int tile_col = 0; tile_col < A.num_col_tiles; ++tile_col)
        {
            int col_base = tile_col * MAX_COLS;
            int cols_this_tile = (col_base + MAX_COLS <= global_num_cols) ? MAX_COLS : (global_num_cols - col_base);

            int tile_idx = tile_row * A.num_col_tiles + tile_col;
            int tile_nnz = A.nnz_counts[tile_idx];
            int tile_nnz_offset = A.nnz_offsets[tile_idx];
            int tile_col_offset = A.col_offsets[tile_idx];

            spmv_csc(
                rows_this_tile,
                cols_this_tile,
                tile_nnz,
                A.row_idx + tile_nnz_offset,
                A.col_ptr + tile_col_offset,
                A.values + tile_nnz_offset,
                x + col_base,
                y + tile_row * MAX_ROWS,
                /*clear_y=*/(tile_col == 0),
                /*write_y=*/(tile_col == A.num_col_tiles - 1));
        }
    }
}
