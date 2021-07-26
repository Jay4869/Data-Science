# don't think this is easy level
class Solution(object):
    def reorderLogFiles(self, logs):
        """
        :type logs: List[str]
        :rtype: List[str]
        """
        digit = []
        letter = []
        for log in logs:
            if log.split()[1].isdigit():
                digit.append(log)
            else:
                letter.append(log)
        letter.sort(key=lambda x: x.split()[0])
        letter.sort(key=lambda x: x.split()[1:])
        return letter + digit


# Time Complexity: nlogn
# Space Complexity: n
