# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):


class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        beg = 1
        end = n
        while beg < end:
            mid = (end + beg) // 2
            if isBadVersion(mid):
                end = mid
            else:
                beg = mid + 1
        return beg