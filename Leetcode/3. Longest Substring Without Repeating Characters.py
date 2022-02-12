class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        chars = collections.Counter()

        left = right = res = 0
        while right < len(s):
            chars[s[right]] += 1
            while chars[s[right]] > 1:
                chars[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res


class Solution:
    def lengthOfLongestSubstring(self, s):
        n = len(s)
        ans = 0
        # mp stores the current index of a character
        mp = {}

        i = 0
        # try to extend the range [i, j]
        for j in range(n):
            if s[j] in mp:
                i = max(mp[s[j]], i)

            ans = max(ans, j - i + 1)
            mp[s[j]] = j + 1

        return ans


# Time Complexity O(n)
# Space O(min(n,m))