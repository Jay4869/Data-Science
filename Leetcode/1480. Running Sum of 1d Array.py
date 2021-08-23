# O(n) O(1)(not considering the space to store it)
class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        sr = [0 for _ in nums]
        sr[0] = nums[0]
        for i in range(1, len(nums)):
            sr[i] = sr[i - 1] + nums[i]

        return sr


# O(n) O(1)


class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in range(1, len(nums)):
            nums[i] = nums[i - 1] + nums[i]

        return nums