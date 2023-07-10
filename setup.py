from distutils.core import setup
setup(
  name = 'camera_trap_al',
  packages = [
    'camera_trap_al',
    'camera_trap_al.deep_learning',
    'camera_trap_al.active_learning_methods',
    'camera_trap_al.active_learning_methods.utils',
    'camera_trap_al.utils'
  ],  
  version = '1.0',  
  license ='CC BY 4.0', 
  description = 'Provides a function that allows a user to label camera trap images as a classification model is being trained',  
  author = 'Gareth Lamb', 
  author_email = 'lambg@hku.hk',
  url = '', # Not set up yet
  keywords = [
    'Animal', 
    'Active Learning', 
    'Pipeline',
    'Species',
    'Hong Kong'
  ],
  install_requires=[   
    'scikit-learn',
    'skl2onnx',
    'torchvision',
    'tqdm',
    'matplotlib',
    'pandas'
  ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Researchers',      
    'Topic :: Software Development :: Build Tools',
    'License :: CC BY 4.0',
    'Programming Language :: Python :: 3.11.3', 
  ],
)