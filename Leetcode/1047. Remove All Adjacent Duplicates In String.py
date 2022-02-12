class Solution(object):
    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        b = []
        for i in s:
            if b and i == b[-1]:
                b.pop()
            else:
                b.append(i)
        return "".join(b)
