class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        bracket_map = {"(": ")", "[": "]", "{": "}"}
        for i in s:
            if i in ("(", "[", "{"):
                stack.append(i)
            elif stack and i == bracket_map[stack[-1]]:
                stack.pop()
            else:
                return False
        return stack == []


# time compelxity O(n)
# Space compelxity O(n)