from __future__ import annotations, unicode_literals

from collections import namedtuple
from typing import Sequence

from ._utils import get_hash, get_hash_int


class DagNode(object):
    """Node in a directed-acyclic graph (DAG).

    Edges:
        DagNodes are connected by edges.  An edge connects two nodes with a label for
        each side:
         - ``upstream_node``: upstream/parent node
         - ``upstream_label``: label on the outgoing side of the upstream node
         - ``downstream_node``: downstream/child node
         - ``downstream_label``: label on the incoming side of the downstream node

        For example, DagNode A may be connected to DagNode B with an edge labelled
        "foo" on A's side, and "bar" on B's side:

           _____               _____
          |     |             |     |
          |  A  >[foo]---[bar]>  B  |
          |_____|             |_____|

        Edge labels may be integers or strings, and nodes cannot have more than one
        incoming edge with the same label.

        DagNodes may have any number of incoming edges and any number of outgoing
        edges.  DagNodes keep track only of their incoming edges, but the entire graph
        structure can be inferred by looking at the furthest downstream nodes and
        working backwards.

    Hashing:
        DagNodes must be hashable, and two nodes are considered to be equivalent if
        they have the same hash value.

        Nodes are immutable, and the hash should remain constant as a result.  If a
        node with new contents is required, create a new node and throw the old one
        away.

    String representation:
        In order for graph visualization tools to show useful information, nodes must
        be representable as strings.  The ``repr`` operator should provide a more or
        less "full" representation of the node, and the ``short_repr`` property should
        be a shortened, concise representation.

        Again, because nodes are immutable, the string representations should remain
        constant.
    """

    def __hash__(self) -> int:
        """Return an integer hash of the node."""
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        """Compare two nodes; implementations should return True if (and only if)
        hashes match.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Return a full string representation of the node."""
        raise NotImplementedError()

    @property
    def short_repr(self) -> str:
        """Return a partial/concise representation of the node."""
        raise NotImplementedError()

    @property
    def incoming_edge_map(self) -> dict[None | int, tuple[KwargReprNode, str, None]]:
        """Provides information about all incoming edges that connect to this node.

        The edge map is a dictionary that maps an ``incoming_label`` to
        ``(outgoing_node, outgoing_label)``.  Note that implicitly, ``incoming_node`` is
        ``self``.  See "Edges" section above.
        """
        raise NotImplementedError()


DagEdge = namedtuple(
    'DagEdge',
    [
        'downstream_node',
        'downstream_label',
        'upstream_node',
        'upstream_label',
        'upstream_selector',
    ],
)


def get_incoming_edges(
    downstream_node: KwargReprNode,
    incoming_edge_map: dict[None | int, tuple[KwargReprNode, str, None]],
) -> list[DagEdge]:
    edges = []
    for downstream_label, upstream_info in list(incoming_edge_map.items()):
        upstream_node, upstream_label, upstream_selector = upstream_info
        edges += [
            DagEdge(
                downstream_node,
                downstream_label,
                upstream_node,
                upstream_label,
                upstream_selector,
            )
        ]
    return edges


def get_outgoing_edges(
    upstream_node: KwargReprNode,
    outgoing_edge_map: dict[None, list[tuple[KwargReprNode, str, None]]],
) -> list[DagEdge]:
    edges = []
    for upstream_label, downstream_infos in sorted(outgoing_edge_map.items()):
        for downstream_info in downstream_infos:
            downstream_node, downstream_label, downstream_selector = downstream_info
            edges += [
                DagEdge(
                    downstream_node,
                    downstream_label,
                    upstream_node,
                    upstream_label,
                    downstream_selector,
                )
            ]
    return edges


class KwargReprNode(DagNode):
    """A DagNode that can be represented as a set of args+kwargs."""

    @property
    def __upstream_hashes(self) -> list[int]:
        hashes = []
        for downstream_label, upstream_info in list(self.incoming_edge_map.items()):
            upstream_node, upstream_label, upstream_selector = upstream_info
            hashes += [
                hash(x)
                for x in [
                    downstream_label,
                    upstream_node,
                    upstream_label,
                    upstream_selector,
                ]
            ]
        return hashes

    @property
    def __inner_hash(self) -> str:
        props = {'args': self.args, 'kwargs': self.kwargs}
        return get_hash(props)

    def __get_hash(self) -> int:
        hashes = self.__upstream_hashes + [self.__inner_hash]
        return get_hash_int(hashes)

    def __init__(
        self,
        incoming_edge_map: dict[int | None, tuple[KwargReprNode, str, str | None]],
        name: str,
        args: Sequence[str | int],
        kwargs: dict[str, str | int | tuple[int, int]],
    ):
        self.__incoming_edge_map = incoming_edge_map
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.__hash = self.__get_hash()

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    @property
    def short_hash(self) -> str:
        return '{:x}'.format(abs(hash(self)))[:12]

    def long_repr(self, include_hash: bool = True) -> str:
        formatted_props = ['{!r}'.format(arg) for arg in self.args]
        formatted_props += ['{}={!r}'.format(key, self.kwargs[key]) for key in sorted(self.kwargs)]
        out = '{}({})'.format(self.name, ', '.join(formatted_props))
        if include_hash:
            out += ' <{}>'.format(self.short_hash)
        return out

    def __repr__(self) -> str:
        return self.long_repr()

    @property
    def incoming_edges(self) -> list[DagEdge]:
        return get_incoming_edges(self, self.incoming_edge_map)

    @property
    def incoming_edge_map(self) -> dict[None | int, tuple[KwargReprNode, str, None]]:
        return self.__incoming_edge_map

    @property
    def short_repr(self) -> str:
        return self.name


def topo_sort(
    downstream_nodes: Sequence[KwargReprNode],
) -> tuple[list[KwargReprNode], dict[KwargReprNode, dict[None, list[tuple[KwargReprNode, str, None]]]]]:
    marked_nodes = []
    sorted_nodes: list[KwargReprNode] = []
    outgoing_edge_maps: dict[KwargReprNode, dict[None, list[tuple[KwargReprNode, str, None]]]] = {}

    def visit(
        upstream_node: KwargReprNode,
        upstream_label: None,
        downstream_node: None,
        downstream_label: None,
        downstream_selector: None = None,
    ) -> None:
        if upstream_node in marked_nodes:
            raise RuntimeError('Graph is not a DAG')

        if downstream_node is not None:
            outgoing_edge_map: dict[None, list[tuple[KwargReprNode, str, None]]] = outgoing_edge_maps.get(
                upstream_node, {}
            )
            outgoing_edge_infos: list[tuple[KwargReprNode, str, None]] = outgoing_edge_map.get(upstream_label, [])
            outgoing_edge_infos += [(downstream_node, downstream_label, downstream_selector)]
            outgoing_edge_map[upstream_label] = outgoing_edge_infos
            outgoing_edge_maps[upstream_node] = outgoing_edge_map

        if upstream_node not in sorted_nodes:
            marked_nodes.append(upstream_node)
            for edge in upstream_node.incoming_edges:
                visit(
                    edge.upstream_node,
                    edge.upstream_label,
                    edge.downstream_node,
                    edge.downstream_label,
                    edge.upstream_selector,
                )
            marked_nodes.remove(upstream_node)
            sorted_nodes.append(upstream_node)

    unmarked_nodes = [(node, None) for node in downstream_nodes]
    while unmarked_nodes:
        upstream_node, upstream_label = unmarked_nodes.pop()
        visit(upstream_node, upstream_label, None, None)
    return sorted_nodes, outgoing_edge_maps
