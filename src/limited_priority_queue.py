import heapq

class LimitedPriorityQueue:
    """
    Custom class for a limited priority queue

    Has very limited functionality due to the scope of this project- i.e. only allows pushing and reading.
    """

    def __init__(self, max_size, reverse=False):
        """
        Constructor for LimitedPriorityQueue

        Args:
            max_size: max size of the queue
            reverse: if True, the queue will be a min heap, else a max heap
        """
        self.max_size = max_size
        self.heap = []
        self.reverse=reverse

    def push(self, item, priority):
        """
        Pushes an item into the queue only if either the queue is not full OR the item has a higher priority than the least priority item in the queue

        Note that if reverse is True, the queue will be a min heap, else a max heap

        Args:
            item: the item to be pushed
            priority: the priority of the item

        Returns:
            None
        """
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (priority, item))
        else:
            min_priority, _ = self.heap[0]
            if self.reverse:
                if priority < min_priority:
                    heapq.heappushpop(self.heap, (priority, item))
            else:
                if priority > min_priority:
                    heapq.heappushpop(self.heap, (priority, item))

    
    def heap_items(self):
        """
        Returns the heap items as a list 
        """
        return [item for _, item in self.heap]