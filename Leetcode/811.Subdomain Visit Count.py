class Solution(object):
    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        hash_map = collections.Counter()
        for cpdomain in cpdomains:
            count, cpdomain = cpdomain.split()
            count = int(count)
            subdomains = cpdomain.split(".")
            for i in range(len(subdomains)):
                hash_map[".".join(subdomains[i:])] += count
        return ["{} {}".format(ct, dom) for dom, ct in hash_map.items()]

# The idea is how to refomulate the sub domain from the cp domains by spliting them and joining them again
# Time Complexity: N
# Space Complexity: N
