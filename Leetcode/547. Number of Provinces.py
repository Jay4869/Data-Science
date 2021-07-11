class Solution(object):
    def dsf(self, r, grid, visited):
        visited.append(r)
        for j in range(len(grid)):
            if grid[r][j] == 1 and j not in visited:
                self.dsf(j, grid, visited)

    def findCircleNum(self, M):
        visited = []
        count = 0
        for row in range(len(M)):
            if row not in visited:
                self.dsf(row, M, visited)
                count += 1
        return count


# O(n^2) O(n)
