OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
rz(0.25*pi) q[1];
rz(0.25*pi) q[2];
cz q[2], q[3];
h q[3];
rz(1.75*pi) q[3];
cx q[1], q[3];
rz(0.25*pi) q[3];
cx q[2], q[3];
rz(1.75*pi) q[3];
cx q[1], q[3];
rz(0.25*pi) q[3];
rz(0.25*pi) q[5];
cz q[7], q[8];
rz(0.25*pi) q[8];
cx q[5], q[8];
rz(0.25*pi) q[8];
cz q[8], q[9];
h q[9];
rz(1.75*pi) q[9];
rz(0.25*pi) q[10];
rz(0.25*pi) q[11];
cz q[10], q[12];
h q[12];
rz(1.75*pi) q[12];
cx q[11], q[12];
rz(0.25*pi) q[12];
cx q[10], q[12];
rz(1.75*pi) q[12];
cx q[10], q[12];
h q[14];
cx q[14], q[12];
cx q[1], q[2];
rz(1.75*pi) q[2];
cz q[2], q[3];
h q[3];
rz(0.25*pi) q[3];
cx q[2], q[4];
cx q[4], q[6];
cx q[6], q[0];
rz(0.75*pi) q[6];
h q[7];
rz(1.75*pi) q[7];
cx q[5], q[7];
rz(0.25*pi) q[7];
cx q[5], q[7];
cx q[8], q[7];
cx q[10], q[11];
rz(1.75*pi) q[12];
cx q[11], q[13];
rz(1.75*pi) q[0];
cx q[3], q[4];
cx q[6], q[0];
rz(1.75*pi) q[7];
cx q[5], q[7];
rz(0.25*pi) q[7];
cx q[8], q[4];
cx q[9], q[7];
cx q[11], q[10];
rz(0.75*pi) q[0];
cx q[0], q[2];
rz(1.75*pi) q[4];
rz(1.75*pi) q[7];
cx q[3], q[7];
rz(0.25*pi) q[7];
cx q[8], q[7];
cx q[8], q[4];
rz(0.25*pi) q[4];
rz(1.75*pi) q[7];
cx q[3], q[7];
rz(0.25*pi) q[7];
h q[7];
rz(0.25*pi) q[7];
cx q[4], q[9];
rz(0.25*pi) q[9];
cx q[8], q[9];
rz(1.75*pi) q[9];
cx q[4], q[9];
rz(0.25*pi) q[9];
h q[9];
rz(0.25*pi) q[9];
cx q[7], q[12];
rz(0.25*pi) q[12];
cx q[11], q[12];
rz(1.75*pi) q[12];
cx q[7], q[12];
rz(0.25*pi) q[12];
h q[12];
rz(0.5*pi) q[12];
cx q[9], q[13];
rz(0.5*pi) q[13];
cx q[3], q[8];
rz(1.75*pi) q[8];
h q[8];
rz(0.5*pi) q[8];
cx q[6], q[8];
rz(1.25*pi) q[8];
cx q[0], q[8];
rz(0.25*pi) q[8];
cx q[6], q[8];
rz(1.75*pi) q[8];
cx q[0], q[8];
rz(0.25*pi) q[8];
h q[8];
cx q[7], q[11];
rz(1.75*pi) q[11];
cx q[7], q[11];
rz(0.25*pi) q[11];
cx q[11], q[14];
rz(1.75*pi) q[14];
cx q[9], q[14];
rz(0.25*pi) q[14];
cx q[11], q[14];
rz(1.75*pi) q[14];
cx q[9], q[14];
rz(0.25*pi) q[14];
h q[14];
cx q[14], q[12];
rz(1.5*pi) q[14];
h q[14];
rz(1.75*pi) q[14];
cx q[0], q[14];
rz(0.25*pi) q[14];
cx q[8], q[6];
cx q[9], q[11];
rz(1.25*pi) q[11];
h q[11];
rz(1.5*pi) q[11];
cx q[9], q[11];
cx q[7], q[11];
rz(0.5*pi) q[11];
rz(1.75*pi) q[12];
cz q[11], q[13];
h q[13];
rz(1.75*pi) q[13];
cx q[0], q[13];
rz(0.25*pi) q[13];
cx q[12], q[14];
rz(0.25*pi) q[14];
h q[14];
rz(0.5*pi) q[14];
h q[11];
rz(0.75*pi) q[11];
cx q[0], q[12];
rz(1.75*pi) q[12];
cx q[11], q[13];
rz(1.75*pi) q[13];
cx q[0], q[13];
rz(0.25*pi) q[13];
h q[13];
cx q[14], q[12];
cx q[0], q[14];
rz(1.5*pi) q[14];
cx q[0], q[11];
rz(1.75*pi) q[11];
cx q[0], q[11];
cx q[13], q[11];
