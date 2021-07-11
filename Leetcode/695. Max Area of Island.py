class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if grid == None or len(grid) == 0:
            return 0
        island = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    space = self.dfs(grid, r, c)
                    island.append(space)
        if len(island) > 0:
            return max(island)
        else:
            return 0

    def dfs(self, grid, r, c):
        nr = len(grid)
        nc = len(grid[0])

        if r < 0 or c < 0 or r >= nr or c >= nc or grid[r][c] == 0:
            return 0
        grid[r][c] = 0
        return (
            1
            + self.dfs(grid, r + 1, c)
            + self.dfs(grid, r - 1, c)
            + self.dfs(grid, r, c + 1)
            + self.dfs(grid, r, c - 1)
        )


# Time Complexity O(NXM)
# Space Complexity O(MxN)