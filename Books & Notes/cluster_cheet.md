### Checking status of the cluster:
* clustat
* clustat -m  -> Display status of and exit
* clustat -s ->  Display status of and exit
* clustat -l -> Use long format for services
* cman_tool status -> Show local record of cluster status
* cman_tool nodes -> Show local record of cluster nodes
* cman_tool nodes -af
* ccs_tool lsnode -> List nodes
* ccs_tool lsfence ->  List fence devices
* group_tool ->  displays the status of fence, dlm and gfs groups
* group_tool ls ->  displays the list of groups and their membership

### Resource Group Control Commands:
* clusvcadm -d -> Disable
* clusvcadm -e -> Enable
* clusvcadm -e -F -> Enable according to failover domain rules
* clusvcadm -e -m -> Enable on
* clusvcadm -r -m -> Relocate to member
* clusvcadm -R ->  Restart a group in place.
* clusvcadm -s -> Stop

### Resource Group Locking (for cluster Shutdown / Debugging):
* clusvcadm -l -> Lock local resource group manager. This prevents resource groups from starting on the local node.
* clusvcadm -S -> Show lock state
* clusvcadm -Z -> Freeze group in place
* clusvcadm -U -> Unfreeze/thaw group
* clusvcadm -u -> Unlock local resource group manager. This allows resource groups to start on the local node.
* clusvcadm -c -> Convalesce (repair, fix) resource group. Attempts to start failed, non-critical resources within a resource group.


### Cluster Command Quick Reference

* clustat -> Display the status of the cluster as viewed from the executing host
* clusvcadm -> Manage services across the cluster.
* clusvcadm -r -m -> Move a service to another node
* clusvcadm -d -> Stop a service
* clusvcadm -e -> Start a service

### Cluster configuration system
* ccs_tool ->  Online management of cluster configuration
* ccs_tool update /etc/cluster/cluster.conf ->  Update the cluster.conf file across the cluster
* cman_tool -> Manage cluster nodes and display the current state of the cluster
* cman_tool status
* fence_node  -> Eject a node from the cluster
* fence_tool dump -> Print fence debug messages