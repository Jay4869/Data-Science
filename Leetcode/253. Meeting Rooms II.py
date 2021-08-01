class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        room = 1
        start_time = []
        end_time = []
        for i in intervals:
            start_time.append(i[0])
            end_time.append(i[1])
        used_rooms = 0
        start_time.sort()
        end_time.sort()
        start_pointer = 0
        end_pointer = 0
        while start_pointer < len(start_time):
            if start_time[start_pointer] >= end_time[end_pointer]:
                used_rooms -= 1
                end_pointer += 1
            used_rooms += 1
            start_pointer += 1
        return used_rooms


# O(nlongn), O(n)