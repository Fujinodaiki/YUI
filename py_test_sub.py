from flask import Flask, url_for, render_template, request
import pytest
from app import app


def test_flask_simple():
    app.config['TESTING'] = True
    result =  app.test_client().post('/',data=dict(
        name='これはテストです'
    ), follow_redirects=True)
    assert result.status_code == 200
    assert b'name' in result.data