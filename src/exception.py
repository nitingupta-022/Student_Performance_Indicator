import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys) :
    '''
    This function is used to get the details of the error that occurred, such as the file name, line number, and error message.
    It takes two arguments: the error message and the error detail (which is the sys module).
    '''

    _, _, exc_tb = error_detail.exc_info()  
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: {file_name}, \nLine number: {line_number}, \nError message: {str(error)}"

    return error_message


class CustomException(Exception) : 
    def __init__(self, error_message, error_detail: sys) : 
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) :
        return self.error_message
    
    
# For testing the CustomException
'''
if __name__ == "__main__" : 
    try : 
        a = 1/0
    except Exception as e :
        logging.info("Division by zero error occurred.")
        raise CustomException(e, sys)
'''