class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def validate(node, low=-math.inf, high=math.inf):
            # Empty trees are valid BSTs.
            if not node:
                return True
            # The current node's value must be between low and high.
            if node.val <= low or node.val >= high:
                return False

            # The left and right subtree must also be valid.
            return validate(node.right, node.val, high) and validate(
                node.left, low, node.val
            )

        return validate(root)