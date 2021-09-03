# Time : O(NlogN)
# Space : O(N)
class Solution(object):
    def highFive(self, items):
        """
        :type items: List[List[int]]
        :rtype: List[List[int]]
        """
        hash_map = {}
        for i in items:
            if i[0] not in hash_map.keys():
                hash_map[i[0]] = [i[1]]
            else:
                hash_map[i[0]].append(i[1])
        result = []
        print(hash_map)
        for i in hash_map:
            hash_map[i].sort()
            result.append([i, sum(hash_map[i][-5:]) // len(hash_map[i][-5:])])
        return result