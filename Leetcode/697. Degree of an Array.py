class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dic_freq = collections.Counter()
        dic_left = {}
        dic_right = {}
        for i, x in enumerate(nums):
            if x not in dic_freq:
                dic_freq[x] = 1
                dic_left[x] = dic_right[x] = i
            else:
                dic_freq[x] += 1
                dic_right[x] = i
        degree = max(dic_freq.values())
        length = len(nums)
        for i in dic_freq:
            if dic_freq[i] == degree:
                length = min(length, dic_right[i] - dic_left[i] + 1)
        print(length)
        return length
