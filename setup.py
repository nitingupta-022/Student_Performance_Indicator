from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str] : 
    '''
    This function will return the list of requirements mentioned in the requirements.txt file
    '''

    requirements = []
    with open(file_path) as file_obj : 
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements : 
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name="ml_project",
    version="0.1.0",
    author="Nitin Gupta",
    author_email="22f3000272@ds.study.iitm.ac.in",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)