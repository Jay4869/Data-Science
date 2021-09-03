# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev_node = None
        curr_node = head
        while curr_node != None:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
        return prev_node

    def reverseList(self, head):
        if head == None or head.next == None:
            return head
        p = reverseList(head.next)
        head.next.next = head
        head.next = None
        return p


# (1) For loop: Time :O(N) Space:O(1)
# (2) Iternative: Time: O(n) Space:O(n)  The extra space comes from implicit stack space due to recursion. The recursion could go up to nn levels deep.
