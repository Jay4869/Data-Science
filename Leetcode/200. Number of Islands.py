class Solution(object):
    def dfs(self, grid, r, c):
        nr = len(grid)
        nc = len(grid[0])

        if r < 0 or c < 0 or r >= nr or c >= nc or grid[r][c] == "0":
            return
        grid[r][c] = "0"
        self.dfs(grid, r + 1, c)
        self.dfs(grid, r - 1, c)
        self.dfs(grid, r, c + 1)
        self.dfs(grid, r, c - 1)

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if grid == None or len(grid) == 0:
            return 0
        num_island = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == "1":
                    num_island += 1
                    self.dfs(grid, r, c)
        return num_island


# Time Complexity O(NXM)
# Space Complexity O(MxN)