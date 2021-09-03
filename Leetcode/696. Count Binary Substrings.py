# my bf solution (exceed time limit) Time:O(N^2)Space:O(N)
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        dic = collections.Counter()
        # check group
        def check_group(a):
            condition1 = a[0 : len(a) / 2] == "0" * (len(a) / 2) and a[
                len(a) / 2 : len(a)
            ] == "1" * (len(a) / 2)
            condition2 = a[0 : len(a) / 2] == "1" * (len(a) / 2) and a[
                len(a) / 2 : len(a)
            ] == "0" * (len(a) / 2)
            if len(a) % 2 == 0 and len(a) >= 2 and (condition1 or condition2):
                return True
            else:
                return False

        # two pointer to get substring
        for i in range(0, len(s)):
            for j in range(i, len(s) + 1):
                if check_group(s[i:j]):
                    dic[s[i:j]] += 1
        return sum(dic.values())


# suggested solution Time: O(N) Space: O(1)
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        ans, prev, cur = 0, 0, 1
        for i in xrange(1, len(s)):
            if s[i - 1] != s[i]:
                ans += min(prev, cur)
                prev, cur = cur, 1
            else:
                cur += 1
        return ans + min(prev, cur)
