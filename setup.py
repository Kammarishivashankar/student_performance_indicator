from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'
def install_packages(req_file_path) -> List[str]:
    """
    parameters : requirements file path
    returns the list of requirements
    """
    requirements = []
    with open(req_file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
    
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements




setup(
    name = "student_performance_indicator",
    version = "0.0.1",
    author = "Shivashankar",
    author_email= "kammarishivashankarr@gmail.com",
    packages = find_packages(),
    install_packages = install_packages('requirements.txt')





)