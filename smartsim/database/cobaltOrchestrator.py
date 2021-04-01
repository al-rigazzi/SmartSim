from ..config import CONFIG
from ..entity import DBNode
from ..error import SmartSimError, SSUnsupportedError
from ..settings import AprunSettings, CobaltBatchSettings, MpirunSettings
from .orchestrator import Orchestrator


class CobaltOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        run_command="aprun",
        account=None,
        queue=None,
        time=None,
        **kwargs,
    ):
        """Initialize an Orchestrator reference for Cobalt based systems

        The orchestrator launches as a batch by default. If batch=False,
        at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The Cobalt orchestrator does not support multiple databases per node.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on
        :type hosts: list[str]
        :param run_command: specify launch binary. Options are ``mpirun`` and ``aprun``
        :type run_command: str
        :param account: account to run batch on
        :type account: str
        :param queue: queue to launch batch in
        :type queue: str
        :param time: walltime for batch 'HH:MM:SS' format
        :type time: str
        """
        super().__init__(
            port, db_nodes=db_nodes, batch=batch, run_command=run_command, **kwargs
        )
        self.batch_settings = self._build_batch_settings(
            db_nodes, batch, account, queue, time
        )
        if hosts:
            self.set_hosts(hosts)
        elif not hosts and run_command == "mpirun":
            raise SmartSimError(
                "hosts argument is required when launching CobaltOrchestrator with OpenMPI"
            )

    def set_cpus(self, num_cpus):
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        # unsure of how to add cpus per task to Cobalt batch settings
        raise NotImplementedError("Cobalt batch doesn't support setting cpus per task")

    def set_walltime(self, walltime):
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running in batch, cannot set walltime")
        self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list):
        """Specify the hosts for the ``CobaltOrchestrator`` to launch on

        :param host_list: list of hosts (compute node names)
        :type host_list: list[str]
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        # TODO check length
        if self.batch:
            self.batch_settings.set_hostlist(host_list)
        for host, db in zip(host_list, self.entities):
            db.set_host(host)

            # Aprun doesn't like settings hosts in batch launch
            if isinstance(db.run_settings, AprunSettings):
                if not self.batch:
                    db.run_settings.set_hostlist([host])
            else:
                db.run_settings.set_hostlist([host])

    def set_batch_arg(self, arg, value):
        """Set a cobalt ``qsub`` argument

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set batch_arg")
        self.batch_settings.batch_args[arg] = value

    def _build_run_settings(self, exe, exe_args, **kwargs):
        run_command = kwargs.get("run_command", "aprun")
        if run_command == "aprun":
            return self._build_aprun_settings(exe, exe_args, **kwargs)
        if run_command == "mpirun":
            return self._build_mpirun_settings(exe, exe_args, **kwargs)
        raise SSUnsupportedError(
            f"CobaltOrchestrator does not support {run_command} as a launch binary"
        )

    def _build_aprun_settings(self, exe, exe_args, **kwargs):
        run_args = kwargs.get("run_args", {})
        run_settings = AprunSettings(exe, exe_args, run_args=run_args)
        run_settings.set_tasks(1)
        run_settings.set_tasks_per_node(1)
        return run_settings

    def _build_mpirun_settings(self, exe, exe_args, **kwargs):
        run_args = kwargs.get("run_args", {})
        run_settings = MpirunSettings(exe, exe_args, run_args=run_args)
        run_settings.set_tasks(1)
        return run_settings

    def _build_batch_settings(self, db_nodes, batch, account, queue, time):
        batch_settings = None
        if batch:
            batch_settings = CobaltBatchSettings(
                nodes=db_nodes, time=time, queue=queue, account=account
            )
        return batch_settings

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator."""
        db_nodes = kwargs.get("db_nodes", 1)
        cluster = not bool(db_nodes < 3)
        if int(db_nodes) == 2:
            raise SSUnsupportedError(
                "CobaltOrchestrator does not support clusters of size 2"
            )
        port = kwargs.get("port", 6379)

        db_conf = CONFIG.redis_conf
        exe = CONFIG.redis_exe
        ip_module = self._get_IP_module_path()
        ai_module = self._get_AI_module()

        # Build DBNode instance for each node listed
        for db_id in range(db_nodes):
            db_node_name = "_".join((self.name, str(db_id)))
            node_exe_args = [db_conf, ai_module, ip_module, "--port", str(port)]
            if cluster:
                node_exe_args += self._get_cluster_args(db_node_name, port)

            run_settings = self._build_run_settings(exe, node_exe_args, **kwargs)
            node = DBNode(db_node_name, self.path, run_settings, [port])
            self.entities.append(node)
        self.ports = [port]
