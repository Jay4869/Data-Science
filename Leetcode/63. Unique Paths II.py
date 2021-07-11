class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        r = len(obstacleGrid)
        c = len(obstacleGrid[0])

        # check if there is a block in (0,0) position
        if obstacleGrid[0][0] == 1:
            return 0
        # for matrix has column or row equal 1
        if r == 1:
            for i in xrange(1, c):
                if obstacleGrid[0][i] == 1:
                    return 0
            return 1
        if c == 1:
            for i in xrange(1, r):
                if obstacleGrid[i][0] == 1:
                    return 0
            return 1
        grid = [[1] * c for _ in range(r)]

        # for matrix has column and row both larger than 2
        for i in xrange(1, r):
            if obstacleGrid[i][0] == 1:
                grid[i][0] = 0
            else:
                grid[i][0] = grid[i - 1][0]
        for j in xrange(1, c):
            if obstacleGrid[0][j] == 1:
                grid[0][j] = 0
            else:
                grid[0][j] = grid[0][j - 1]

        for col in xrange(1, c):
            for row in xrange(1, r):
                if obstacleGrid[row][col] == 1:
                    grid[row][col] = 0
                else:
                    grid[row][col] = grid[row - 1][col] + grid[row][col - 1]
        return grid[r - 1][c - 1]
