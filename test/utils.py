from colorama import Fore ,Style
class FileExistsNotFound(Exception):
    def __init__(self, msg: str):
        if not isinstance(msg, str):
            raise TypeError("message harus dalam bentuk string")
        else:
            self.msg = Fore.RED + msg + Style.RESET_ALL
    def __str__(self):
        return self.msg


class FormatFileError(Exception):
    def __init__(self,msg:str):
        if not isinstance(msg,str):
            raise TypeError("message harus dalam bentuk string")
        else:
            self.msg = Fore.YELLOW + self.msg + Style.RESET_ALL
    
    def __str__(self):
        return self.msg


