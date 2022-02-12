class MinStack(list):
    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        m = min(val, self[-1][1] if self else float("inf"))
        self.append((val, m))

    def pop(self):
        """
        :rtype: None
        """
        return list.pop(self)[0]

    def top(self):
        """
        :rtype: int
        """
        return self[-1][0]

    def getMin(self):
        """
        :rtype: int
        """
        return self[-1][1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()