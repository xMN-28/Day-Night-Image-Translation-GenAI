import json

from daynight.download_reference import ranged_download


def test_range_download_preserves_received_prefix_and_orders_parallel_chunks(tmp_path, monkeypatch):
    body = bytes(range(251)) * 700

    class Response:
        status_code = 206

        def __init__(self, start, end):
            self.start, self.end = start, end
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(body)}", "ETag": "known"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            yield body[self.start : self.end + 1]

    def get(url, headers, **kwargs):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return Response(start, end)

    monkeypatch.setattr("daynight.download_reference.requests.get", get)
    output = tmp_path / "checkpoint.pkl"
    output.with_suffix(".download").write_bytes(body[:9000])
    ranged_download("https://official.example/weights", output, workers=4)
    assert output.read_bytes() == body
    state = json.loads(output.with_suffix(".ranges.json").read_text())
    assert state["prefix_bytes"] == 9000
    assert state["complete"]


def test_resume_uses_manifest_not_preallocated_file_length(tmp_path, monkeypatch):
    body = bytes(range(32))
    requests_seen = []

    class Response:
        status_code = 206

        def __init__(self, start, end):
            self.start, self.end = start, end
            self.headers = {"Content-Range": f"bytes {start}-{end}/32", "ETag": "known"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            yield body[self.start : self.end + 1]

    def get(url, headers, **kwargs):
        requests_seen.append(headers["Range"])
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return Response(start, end)

    monkeypatch.setattr("daynight.download_reference.requests.get", get)
    output = tmp_path / "checkpoint.pkl"
    output.with_suffix(".download").write_bytes(body[:12] + bytes(8) + body[20:])
    output.with_suffix(".ranges.json").write_text(
        json.dumps(
            {
                "url": "https://official.example/weights",
                "size": 32,
                "etag": "known",
                "prefix_bytes": 8,
                "ranges": [
                    {"start": 8, "end": 20, "done": 4},
                    {"start": 20, "end": 32, "done": 12},
                ],
            }
        )
    )
    ranged_download("https://official.example/weights", output, workers=2)
    assert requests_seen == ["bytes=0-0", "bytes=12-19"]
    assert output.read_bytes() == body
