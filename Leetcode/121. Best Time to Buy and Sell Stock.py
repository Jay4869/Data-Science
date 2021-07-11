class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        min_price = 99999
        max_profit = 0
        for i in prices:
            min_price = min(i, min_price)
            max_profit = max(max_profit, i - min_price)
        return max_profit


# Time Cmoplexity is just O(N)
# Space compexity is O(1)