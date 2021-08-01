class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if digits == "":
            return []
        dict = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"],
        }
        current_values = [""]
        for digit in digits:
            next_values = []
            for val in current_values:
                for ch in dict[digit]:
                    next_values.append(val + ch)
            current_values = next_values
        return current_values


# O(3^n*4^m)