/**
 * Generic user creation / update script.
 *
 * Usage:
 *   npx tsx scripts/create-user.ts <email> <password> <role> [name]
 *
 * Role must be ADMIN or TEACHER.
 *
 * Examples:
 *   npx tsx scripts/create-user.ts teacher@university.edu secret123 TEACHER "Dr. Jane Smith"
 *   npx tsx scripts/create-user.ts admin@university.edu secret123 ADMIN   "John Doe"
 */
import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcrypt';

const prisma = new PrismaClient();

async function main() {
  const [email, password, role, name] = process.argv.slice(2);

  if (!email || !password || !role) {
    console.error('Usage: npx tsx scripts/create-user.ts <email> <password> <ADMIN|TEACHER> [name]');
    process.exit(1);
  }

  if (role !== 'ADMIN' && role !== 'TEACHER') {
    console.error('Role must be ADMIN or TEACHER');
    process.exit(1);
  }

  const password_hash = await bcrypt.hash(password, 10);

  const user = await prisma.user.upsert({
    where:  { email },
    update: { password_hash, role: role as 'ADMIN' | 'TEACHER', name: name ?? null },
    create: { email, password_hash, role: role as 'ADMIN' | 'TEACHER', name: name ?? null },
  });

  console.log(`✓  ${user.email}  [${user.role}]${user.name ? '  — ' + user.name : ''}`);
}

main()
  .catch(e => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
