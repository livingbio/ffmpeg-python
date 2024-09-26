from __future__ import unicode_literals

from typing import Any

from .nodes import (
    FilterableStream,
    GlobalNode,
    InputNode,
    MergeOutputsNode,
    OutputNode,
    OutputStream,
    Stream,
    filter_operator,
    output_operator,
)


def input(filename: str, **kwargs: Any) -> Stream:
    """Input file URL (ffmpeg ``-i`` option)

    Any supplied kwargs are passed to ffmpeg verbatim (e.g. ``t=20``,
    ``f='mp4'``, ``acodec='pcm'``, etc.).

    To tell ffmpeg to read from stdin, use ``pipe:`` as the filename.

    Official documentation: `Main options <https://ffmpeg.org/ffmpeg.html#Main-options>`__
    """
    kwargs['filename'] = filename
    fmt = kwargs.pop('f', None)
    if fmt:
        if 'format' in kwargs:
            raise ValueError("Can't specify both `format` and `f` kwargs")
        kwargs['format'] = fmt
    return InputNode(input.__name__, kwargs=kwargs).stream()


@output_operator()
def global_args(stream: OutputStream, *args: str) -> Stream:
    """Add extra global command-line argument(s), e.g. ``-progress``."""
    return GlobalNode(stream, global_args.__name__, args).stream()


@output_operator()
def overwrite_output(stream: OutputStream) -> Stream:
    """Overwrite output files without asking (ffmpeg ``-y`` option)

    Official documentation: `Main options <https://ffmpeg.org/ffmpeg.html#Main-options>`__
    """
    return GlobalNode(stream, overwrite_output.__name__, ['-y']).stream()


@output_operator()
def merge_outputs(*streams: OutputStream) -> Stream:
    """Include all given outputs in one ffmpeg command line"""
    return MergeOutputsNode(streams, merge_outputs.__name__).stream()


@filter_operator()
def output(*streams_and_filename_: str | FilterableStream, **kwargs: Any) -> OutputStream:
    """Output file URL

    Syntax:
        `ffmpeg.output(stream1[, stream2, stream3...], filename, **ffmpeg_args)`

    Any supplied keyword arguments are passed to ffmpeg verbatim (e.g.
    ``t=20``, ``f='mp4'``, ``acodec='pcm'``, ``vcodec='rawvideo'``,
    etc.).  Some keyword-arguments are handled specially, as shown below.

    Args:
        video_bitrate: parameter for ``-b:v``, e.g. ``video_bitrate=1000``.
        audio_bitrate: parameter for ``-b:a``, e.g. ``audio_bitrate=200``.
        format: alias for ``-f`` parameter, e.g. ``format='mp4'``
            (equivalent to ``f='mp4'``).

    If multiple streams are provided, they are mapped to the same
    output.

    To tell ffmpeg to write to stdout, use ``pipe:`` as the filename.

    Official documentation: `Synopsis <https://ffmpeg.org/ffmpeg.html#Synopsis>`__
    """
    streams_and_filename = list(streams_and_filename_)
    if 'filename' not in kwargs:
        if not isinstance(streams_and_filename[-1], str):
            raise ValueError('A filename must be provided')
        kwargs['filename'] = streams_and_filename.pop(-1)
    streams = streams_and_filename

    fmt = kwargs.pop('f', None)
    if fmt:
        if 'format' in kwargs:
            raise ValueError("Can't specify both `format` and `f` kwargs")
        kwargs['format'] = fmt
    stream = OutputNode(tuple(k for k in streams if isinstance(k, Stream)), output.__name__, kwargs=kwargs).stream()
    assert isinstance(stream, OutputStream)
    return stream


__all__ = ['input', 'merge_outputs', 'output', 'overwrite_output']
