from flask import Flask, jsonify, request

from routing import calculate_route

app = Flask(__name__)

@app.route('/api/route', methods=['GET'])
def get_route():
    data = request.get_json(force=True)

    origin_lat = data["origin_lat"]
    origin_lon = data["origin_lon"]
    dest_lat = data["dest_lat"]
    dest_lon = data["dest_lon"]
    mode = data["mode"]

    result = calculate_route(origin_lat, origin_lon, dest_lat, dest_lon, mode)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
