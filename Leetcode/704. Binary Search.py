# N(logN)
class Solution:
    def search(self, nums, target):
        beg, end = 0, len(nums) - 1
        while beg <= end:
            mid = (beg + end) / 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                beg = mid + 1
            else:
                end = mid - 1
        return -1
