"""Compile and exercise the actual C++ TCP implementation against slow servers."""
from pathlib import Path
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest

RUNTIME_ROOT=Path(os.environ.get("MOZCAI_RUNTIME_ROOT",
    str(Path(__file__).resolve().parents[3]/"Mozc-Ai")))
SOURCE=RUNTIME_ROOT/"mozc_compat/rerank_rewriter.cc"


class TcpDeadlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text=SOURCE.read_text()
        start=text.index("#ifdef _WIN32\nconstexpr RerankSocket")
        end=text.index("\n}  // namespace",start)
        cls.directory=tempfile.TemporaryDirectory()
        root=Path(cls.directory.name)
        code='''
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cerrno>
#include <string>
#include <mutex>
#include <iostream>
using RerankSocket=int;
'''+text[start:end]+'''
int main(int argc, char** argv) {
  std::string response;
  auto start=std::chrono::steady_clock::now();
  bool ok=TcpExchange("127.0.0.1",std::stoi(argv[1]),"{}\\n",200,&response);
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cout << ok << " " << ms << " " << response;
}
'''
        src=root/"client.cc";src.write_text(code)
        cls.binary=root/"client"
        subprocess.run(["g++","-std=c++17","-O2",str(src),"-o",str(cls.binary)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def exchange(self,chunks,delay):
        with socket.socket() as server:
            server.bind(("127.0.0.1",0));server.listen()
            def respond():
                try:
                    conn,_=server.accept()
                    with conn:
                        conn.recv(1024)
                        for chunk in chunks:
                            time.sleep(delay);conn.sendall(chunk)
                except OSError:pass
            thread=threading.Thread(target=respond,daemon=True);thread.start()
            result=subprocess.run([str(self.binary),str(server.getsockname()[1])],capture_output=True,text=True,timeout=2)
            fields=result.stdout.split(" ",2)
            thread.join(timeout=1)
            return fields[0]=="1",float(fields[1]),fields[2] if len(fields)>2 else ""

    def test_complete_response(self):
        ok,ms,response=self.exchange([b'{"ok":true}\n'],.01)
        self.assertTrue(ok);self.assertLess(ms,200);self.assertEqual(response,'{"ok":true}')

    def test_partial_responses_cannot_extend_total_deadline(self):
        ok,ms,_=self.exchange([b'{' for _ in range(8)],.07)
        self.assertFalse(ok);self.assertGreater(ms,180);self.assertLess(ms,300)

    def test_silent_server_returns_at_deadline(self):
        ok,ms,_=self.exchange([b'{}\n'],.5)
        self.assertFalse(ok);self.assertGreater(ms,180);self.assertLess(ms,300)


if __name__=="__main__":unittest.main()
