class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        first_dip = 0
        larger_than_dip = 0
        # find the first numbers that are smaller than the privous number from right to left
        for i in xrange(l - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                first_dip = i - 1
                break
        # check whether the numbers are in decreasing order from left to right
        # if so, we don't have next permutation
        if first_dip == 0 and l > 1 and nums[0] > nums[1]:
            return nums.sort()

        # search for the numbers on the right side of the number in first_dip position
        # find the smallest number that is larger than number in first_dip position
        for i in xrange(l - 1, first_dip, -1):
            if nums[i] > nums[first_dip]:
                larger_than_dip = i
                break
        # swap numbers in larger_than_dip and first_dip positions
        nums[larger_than_dip], nums[first_dip] = nums[first_dip], nums[larger_than_dip]
        # reverse the numbers in the right side of first_dip position
        sub_str = nums[first_dip + 1 :]
        nums[first_dip + 1 :] = sub_str[::-1]
        return nums
