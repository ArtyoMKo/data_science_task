class GenericException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class ExceededMaxIterationsError(GenericException):
    def __init__(self, msg):
        super().__init__(msg)


class WrongModelStructure(Exception):
    def __init__(self, msg):
        super().__init__(msg)
