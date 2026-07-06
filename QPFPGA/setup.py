from setuptools import setup, find_packages

setup(
    name="qpfpga",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.4,<2.0.0",
        "scipy>=1.13.0",
        "cvxpy==1.7.5",
        "clarabel>=0.5.0",
        "osqp>=0.6.2",
        "scs>=3.2.4.post1",
        "qdldl>=0.1.5",
        "ecos>=2.0.10",
        "wheel",
        "pynq==3.0.1"
    ],
    author="Alejandro Romo Iribarren",
    description="QP Accelerator Solver for FPGA",
    python_requires=">=3.9",
)