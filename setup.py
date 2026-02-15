"""
The Time-Series Forecasting Transformer (TSFT) with Segment-wise Mixture-of-Experts (Seg-MoE)
"""

from setuptools import setup, find_packages

setup(
    name='segmoe_forecast',
    version='2.0.1',
    description='Long-term Time Series Forecasting with Segment-wise Mixture-of-Experts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Evandro S. Ortigossa',
    url='https://github.com/evortigosa/segmoe_forecast',
    packages=find_packages(include=["segmoe_forecast", "segmoe_forecast.*"]),
)
