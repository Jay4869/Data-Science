# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        nodes_in_B = set()
        while headB is not None:
            nodes_in_B.add(headB)
            headB = headB.next

        while headA is not None:
            if headA in nodes_in_B:
                return headA
            headA = headA.next
        return None


# O(N+M)
# O(M)