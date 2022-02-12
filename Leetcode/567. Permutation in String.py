# O(2*l1+(l2-l1)*l1)
# O(l1+l2)
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        d1 = collections.Counter(s1)
        d2 = collections.Counter(s2[: len(s1)])
        for i in xrange(len(s1), len(s2)):
            if d1 == d2:
                return True
            d2[s2[i]] += 1
            d2[s2[i - len(s1)]] -= 1
            if d2[s2[i - len(s1)]] == 0:
                del d2[s2[i - len(s1)]]
        return d1 == d2
