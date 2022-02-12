class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        row = [[1]]
        for r in xrange(1, numRows):
            row.append([1] * (r + 1))
            for c in xrange(1, r):
                row[r][c] = row[r - 1][c] + row[r - 1][c - 1]
        return row