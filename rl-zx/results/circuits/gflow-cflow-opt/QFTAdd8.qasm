OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(0.375*pi) q[2];
rz(0.4375*pi) q[3];
rz(0.46875*pi) q[4];
rz(0.9921875*pi) q[6];
cx q[6], q[7];
rz(1.5*pi) q[7];
cz q[8], q[9];
rz(0.25*pi) q[9];
cz q[9], q[10];
rz(0.375*pi) q[10];
rz(0.9375*pi) q[11];
cz q[10], q[11];
rz(0.96875*pi) q[12];
rz(0.984375*pi) q[13];
rz(0.5*pi) q[14];
rz(0.99609375*pi) q[15];
h q[8];
rz(1.75*pi) q[8];
cx q[9], q[8];
h q[9];
rz(1.75*pi) q[9];
cx q[10], q[8];
cx q[10], q[9];
cx q[11], q[9];
rz(1.875*pi) q[8];
rz(1.875*pi) q[9];
cx q[10], q[8];
h q[10];
rz(1.75*pi) q[10];
cx q[11], q[8];
cx q[12], q[11];
rz(1.9375*pi) q[8];
rz(1.5*pi) q[11];
cx q[11], q[10];
cx q[11], q[9];
cx q[11], q[8];
cx q[12], q[11];
cx q[13], q[12];
rz(1.96875*pi) q[8];
rz(1.9375*pi) q[9];
rz(1.875*pi) q[10];
h q[11];
rz(1.75*pi) q[11];
cx q[12], q[11];
cx q[12], q[10];
cx q[12], q[9];
cx q[12], q[8];
h q[12];
rz(0.5*pi) q[12];
cx q[13], q[12];
cx q[14], q[13];
rz(1.984375*pi) q[8];
rz(1.96875*pi) q[9];
rz(1.9375*pi) q[10];
rz(1.875*pi) q[11];
rz(1.25*pi) q[12];
cx q[13], q[12];
cx q[13], q[11];
cx q[13], q[10];
cx q[13], q[9];
cx q[13], q[8];
h q[13];
rz(0.5*pi) q[13];
cx q[14], q[13];
cx q[15], q[14];
rz(1.9921875*pi) q[8];
rz(1.984375*pi) q[9];
rz(1.96875*pi) q[10];
rz(1.9375*pi) q[11];
rz(1.875*pi) q[12];
rz(1.25*pi) q[13];
cx q[14], q[8];
cx q[14], q[12];
cx q[14], q[13];
cx q[14], q[9];
cx q[14], q[10];
cx q[14], q[11];
rz(1.5*pi) q[14];
cx q[15], q[14];
rz(1.99609375*pi) q[8];
rz(1.9921875*pi) q[9];
cx q[2], q[9];
rz(1.984375*pi) q[10];
rz(1.96875*pi) q[11];
rz(1.9375*pi) q[12];
cx q[5], q[12];
rz(1.875*pi) q[13];
rz(0.9921875*pi) q[14];
h q[14];
rz(1.75*pi) q[14];
cx q[15], q[8];
cx q[15], q[12];
cx q[15], q[13];
cx q[15], q[9];
cx q[15], q[10];
cx q[15], q[11];
cx q[15], q[14];
cx q[8], q[1];
cz q[0], q[8];
rz(1.75*pi) q[9];
cx q[2], q[9];
cz q[2], q[10];
cx q[3], q[10];
rz(1.75*pi) q[10];
cx q[4], q[10];
cx q[3], q[10];
rz(1.875*pi) q[10];
cz q[3], q[11];
cx q[4], q[11];
rz(1.75*pi) q[11];
rz(1.75*pi) q[12];
cz q[4], q[12];
cx q[5], q[12];
cx q[6], q[12];
rz(1.875*pi) q[12];
cx q[7], q[12];
rz(1.9375*pi) q[12];
cz q[5], q[13];
rz(1.125*pi) q[13];
cx q[6], q[13];
rz(1.75*pi) q[13];
cx q[7], q[13];
rz(1.875*pi) q[13];
cx q[15], q[13];
cx q[15], q[12];
rz(1.75*pi) q[1];
cx q[4], q[5];
cx q[8], q[1];
cx q[2], q[8];
rz(1.875*pi) q[8];
cx q[3], q[8];
cx q[2], q[8];
rz(1.9375*pi) q[8];
cx q[4], q[8];
cx q[3], q[8];
rz(1.96875*pi) q[8];
cx q[5], q[8];
rz(1.984375*pi) q[8];
cx q[5], q[10];
rz(1.9375*pi) q[10];
cx q[5], q[11];
rz(1.875*pi) q[11];
rz(1.9375*pi) q[12];
rz(1.875*pi) q[13];
rz(0.25*pi) q[1];
cz q[1], q[9];
cx q[3], q[9];
rz(1.875*pi) q[9];
cx q[4], q[9];
cx q[3], q[9];
rz(1.9375*pi) q[9];
cx q[5], q[9];
rz(1.96875*pi) q[9];
cx q[4], q[5];
cx q[6], q[5];
cx q[5], q[8];
rz(1.9921875*pi) q[8];
cx q[7], q[8];
rz(1.99609375*pi) q[8];
cx q[5], q[9];
rz(1.984375*pi) q[9];
cx q[7], q[9];
rz(1.9921875*pi) q[9];
cx q[5], q[10];
rz(1.96875*pi) q[10];
cx q[7], q[10];
rz(1.984375*pi) q[10];
cx q[5], q[11];
rz(1.9375*pi) q[11];
cx q[7], q[11];
rz(1.96875*pi) q[11];
cx q[15], q[11];
cx q[15], q[10];
cx q[15], q[9];
cx q[15], q[8];
cx q[6], q[5];
cx q[6], q[7];
rz(0.99609375*pi) q[7];
rz(1.99609375*pi) q[8];
rz(1.9921875*pi) q[9];
rz(1.984375*pi) q[10];
rz(1.96875*pi) q[11];
cx q[7], q[14];
rz(1.75*pi) q[14];
cz q[6], q[14];
cx q[15], q[14];
cx q[7], q[15];
rz(1.99609375*pi) q[15];
rz(0.484375*pi) q[5];
cz q[4], q[5];
rz(0.25*pi) q[14];
cx q[15], q[14];
rz(0.25*pi) q[14];
h q[14];
cx q[14], q[13];
cx q[14], q[12];
cx q[14], q[11];
cx q[14], q[10];
cx q[14], q[9];
cx q[14], q[8];
cx q[15], q[14];
rz(1.9921875*pi) q[8];
rz(1.984375*pi) q[9];
rz(1.96875*pi) q[10];
rz(1.9375*pi) q[11];
rz(1.875*pi) q[12];
rz(1.75*pi) q[13];
h q[13];
rz(1.484375*pi) q[13];
cx q[13], q[14];
cx q[14], q[12];
cx q[14], q[11];
cx q[14], q[10];
cx q[14], q[9];
cx q[14], q[8];
rz(1.5*pi) q[14];
cx q[13], q[14];
rz(0.9921875*pi) q[14];
rz(1.984375*pi) q[8];
rz(1.96875*pi) q[9];
rz(1.9375*pi) q[10];
rz(1.875*pi) q[11];
rz(1.25*pi) q[12];
cx q[13], q[12];
cx q[13], q[8];
rz(1.8125*pi) q[12];
h q[12];
cx q[12], q[11];
cx q[12], q[10];
cx q[12], q[9];
cx q[13], q[12];
rz(1.9375*pi) q[9];
rz(1.875*pi) q[10];
rz(1.75*pi) q[11];
cx q[12], q[8];
cx q[12], q[11];
cx q[12], q[10];
cx q[12], q[9];
rz(0.46875*pi) q[12];
rz(1.96875*pi) q[8];
rz(1.4765625*pi) q[9];
rz(1.453125*pi) q[10];
rz(1.40625*pi) q[11];
h q[11];
cx q[11], q[8];
cx q[11], q[10];
cx q[11], q[9];
rz(0.4375*pi) q[11];
cx q[12], q[8];
rz(1.9375*pi) q[8];
rz(1.875*pi) q[9];
rz(1.75*pi) q[10];
h q[10];
cx q[10], q[8];
cx q[10], q[9];
rz(0.375*pi) q[10];
cx q[11], q[8];
cx q[11], q[9];
cz q[10], q[11];
rz(1.875*pi) q[8];
rz(1.75*pi) q[9];
h q[9];
cx q[9], q[8];
rz(0.25*pi) q[9];
cx q[10], q[8];
cz q[9], q[10];
rz(1.75*pi) q[8];
cx q[9], q[8];
rz(1.48828125*pi) q[8];
h q[8];
