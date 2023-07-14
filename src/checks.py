import os
thisdir=os.getcwd()
def check_if_models_exist():
    if os.path.exists(f'{thisdir}/models/'):
        return True
    else:
        return False