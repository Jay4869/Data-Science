class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        temp = []
        curr = head
        while curr is not None:
            temp.append(curr.val)
            curr = curr.next
        return temp[::-1] == temp


# Time O(N) Space O(N)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None:
            return True

        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse(first_half_end.next)

        first_node = head
        second_node = second_half_start
        while second_node is not None:
            if first_node.val != second_node.val:
                return False
            first_node = first_node.next
            second_node = second_node.next

        return True

    def end_of_first_half(self, head):
        slow = head
        fast = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse(self, head):
        prev_node = None
        curr_node = head
        while curr_node is not None:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
        return prev_node


# O(N) and O(1)