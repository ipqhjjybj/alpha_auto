# coding:utf-8

from setuptools import find_packages, setup
# or
# from distutils.core import setup

long_description = "alpha_auto"
setup(
    name='tumbler',  # 包名字
    version='1.0.1',  # 包版本
    author="ipqhjjybj",
    author_email='250657661@qq.com',  # 作者邮箱
    license="MIT LICENSE",
    url='',  # 包的主页
    description='One Alpha Produce script',  # 简单描述
    long_description=long_description,
    include_package_data=True,
    # packages=['tumbler'],
    packages=find_packages(exclude=["example"]),  # 包
    classifiers=[
        # 发展时期,常见的如下
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # 开发的目标用户
        'Intended Audience :: Developers',

        # 属于什么类型
        'Topic :: Software Development :: Build Tools',

        # 目标 Python 版本
        'Programming Language :: Python :: 3.6',
    ]
)
