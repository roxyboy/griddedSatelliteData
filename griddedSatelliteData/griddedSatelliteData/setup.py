from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='griddedSatelliteData',
      version='0.1',
      description='Statistical analysis for gridded satellite data',
      author='Takaya Uchida',
      author_email='takaya@ldeo.columbia.edu',
      license='LDEO',
      packages=['griddedSatelliteData'],
      install_requires=[
          'numpy','scipy','xarray','gsw','pynio'
      ],
      zip_safe=False)
