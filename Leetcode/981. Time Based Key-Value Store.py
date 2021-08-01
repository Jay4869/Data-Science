class TimeMap(object):
    def __init__(self):
        self.dic = collections.defaultdict(list)

    def set(self, key, value, timestamp):
        self.dic[key].append([timestamp, value])

    def get(self, key, timestamp):
        arr = self.dic[key]
        n = len(arr)

        left = 0
        right = n

        while left < right:
            mid = (left + right) / 2
            if arr[mid][0] <= timestamp:
                left = mid + 1
            elif arr[mid][0] > timestamp:
                right = mid

        return "" if right == 0 else arr[right - 1][1]


## O(1)  o(logN)
# O(n)