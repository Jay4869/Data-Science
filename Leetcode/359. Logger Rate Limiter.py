class Logger(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._dic = {}

    def shouldPrintMessage(self, timestamp, message):
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        :type timestamp: int
        :type message: str
        :rtype: bool
        """
        if message in self._dic.keys():
            p_time = self._dic[message]
        else:
            self._dic[message] = timestamp
            return True

        if timestamp - p_time >= 10:
            self._dic[message] = timestamp
            return True
        else:
            return False


# Your Logger object will be instantiated and called as such:
# obj = Logger()
# param_1 = obj.shouldPrintMessage(timestamp,message)