import json
from flask import Flask
from flask_sockets import Sockets

app = Flask(__name__)
sockets = Sockets(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@sockets.route('/media')
def echo(ws):
    has_seen_media = False
    message_count = 0
    while not ws.closed:
        message = ws.receive()
        if message is None:
            print("No message received...")
            continue
        data = json.loads(message)
        if data['event'] == "connected":
            app.logger.info("Connected Message received: {}".format(message))
        if data['event'] == "start":
            app.logger.info("Start Message received: {}".format(message))
        if data['event'] == "media":
            if not has_seen_media:
                app.logger.info("Media message: {}".format(message))
                payload = data['media']['payload']
                app.logger.info("Payload is: {}".format(payload))
                chunk = base64.b64decode(payload)
                app.logger.info("That's {} bytes".format(len(chunk)))
                app.logger.info("Additional media messages from WebSocket are being suppressed....")
                has_seen_media = True
        if data['event'] == "closed":
            app.logger.info("Closed Message received: {}".format(message))
            break
        message_count += 1

    print("Connection closed. Received a total of {} messages".format(message_count))

if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', 5001), app, handler_class=WebSocketHandler)
    print("Server listening on: http://localhost:" + str(5001))
    server.serve_forever()
