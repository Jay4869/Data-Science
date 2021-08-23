# O(n) O(n)
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n

        dp = [0 for _ in range(n)]
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] += dp[i - 1] + dp[i - 2]
        return dp[-1]


# O(n) O(1)
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n

        step1 = 1
        step2 = 2
        for i in range(2, n):
            step3 = step1 + step2
            step1 = step2
            step2 = step3
        return step2