from unittest.mock import MagicMock, patch, call
import propose.preprocessing.rat7m.format as pp


@patch("propose.preprocessing.rat7m.format.ffmpeg")
@patch("propose.preprocessing.rat7m.format.Path")
@patch("propose.preprocessing.rat7m.format.os")
@patch("propose.preprocessing.rat7m.format.tqdm")
def test_convert_movies_to_images(tqdm_mock, os_mock, Path_mock, ffmpeg_mock):
    dirname = "/"
    datakey = "s0-d0"

    os_mock.listdir = MagicMock(return_value=["s0-d0-camera1-0.mp4"])

    tqdm_mock.side_effect = lambda x, *args, **kwargs: x

    pp.convert_movies_to_images(dirname, datakey)

    assert Path_mock.mock_calls[0] == call(
        f"{dirname + datakey}/images/{datakey}-camera1-0"
    )

    assert ffmpeg_mock.mock_calls[0] == call.input(
        f"{dirname + datakey}/movies/{datakey}-camera1-0.mp4"
    )
    assert ffmpeg_mock.mock_calls[1] == call.output(
        ffmpeg_mock.input(),
        f"{dirname + datakey}/images/{datakey}-camera1-0/{datakey}-camera1-%05d.jpg",
        format="image2",
        vcodec="mjpeg",
    )
    assert ffmpeg_mock.mock_calls[2] == call.output().run(quiet=False)
